"""A-5 MIMIC: say a word in a TARGET VOICE given a few reference clips of that voice.

The goal's "mimic voices" piece. Speech factors into CONTENT (which word / how it's pronounced)
and SPEAKER (voice timbre). Mastery = recombine them: synthesize a word never paired with a voice,
in that voice. The model extracts a speaker embedding from a few reference clips (few-shot, no
gradient updates per voice) and conditions a char->spectrogram synthesizer on it.

Tested on HELD-OUT (word x voice) cells -- a word never heard in the target voice:
  word correctness : synthesized audio -> frozen word-listener transcribes it right (pronunciation
                     survived voice transfer)
  voice match      : synthesized audio's speaker-embedding nearest neighbour is the TARGET voice
                     (it actually mimicked, vs defaulting to an average voice)

  python -m thinking.mimic --build
  python -m thinking.mimic --selftest
  python -m thinking.mimic --train --steps 6000 --out runs/a5_mimic.json
"""
import argparse
import json
import os
import subprocess
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .vocoder import stft_logmag, griffin_lim, write_wav, N_BINS, N_FRAMES, SR
from .pronounce import WORDS, encode_chars, MAX_CHARS, CHARS, C2I, CharTTS

DEV = get_device()
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mimic")
VOICES = ("Samantha", "Daniel", "Albert", "Karen", "Fred", "Moira")
RATE = 180
# held-out (word_idx, voice_idx) cells: a word never heard in that voice
def _holdout_cells():
    cells = set()
    for vi in range(len(VOICES)):
        for k in range(8):
            cells.add(((vi * 13 + k * 7) % len(WORDS), vi))
    return cells
HOLDOUT_CELLS = _holdout_cells()


def _available_voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=30)
    names = {ln.split()[0] for ln in out.stdout.splitlines() if ln.strip()}
    return [v for v in VOICES if v in names]


def build_bank(root=ROOT):
    os.makedirs(root, exist_ok=True)
    voices = _available_voices()
    if len(voices) < 3:
        raise RuntimeError(f"need >=3 voices, found {voices}")
    manifest = {"voices": voices, "words": list(WORDS)}
    for vi, voice in enumerate(voices):
        for w in WORDS:
            p = os.path.join(root, f"{w}__{vi}.wav")
            if not os.path.exists(p):
                subprocess.run(["say", "-v", voice, "-r", str(RATE), "-o", p,
                                "--file-format=WAVE", f"--data-format=LEI16@{SR}", w],
                               check=True, timeout=60)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"mimic bank: {len(WORDS)} words x {len(voices)} voices, {len(HOLDOUT_CELLS)} held-out cells")
    return manifest


def load_bank(root=ROOT):
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    voices = manifest["voices"]
    specs = {}
    for vi in range(len(voices)):
        for w in manifest["words"]:
            p = os.path.join(root, f"{w}__{vi}.wav")
            wv = wave.open(p)
            x = np.frombuffer(wv.readframes(wv.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            nz = np.where(np.abs(x) > 0.01)[0]
            if len(nz):
                x = x[max(0, nz[0] - 256):nz[-1] + 256]
            specs[(w, vi)] = stft_logmag(x)
    return manifest, specs


class SpeakerEncoder(nn.Module):
    """Reference spectrograms -> a voice embedding (mean over a few clips)."""

    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, d), nn.LayerNorm(d))

    def forward(self, refs):                               # (B, R, 1, N_BINS, N_FRAMES)
        B, R = refs.shape[:2]
        z = self.proj(self.conv(refs.reshape(B * R, 1, N_BINS, N_FRAMES)))
        return z.reshape(B, R, -1).mean(1)                 # (B, d)


class MimicTTS(nn.Module):
    """CharTTS conditioned on a speaker embedding from reference clips -> spectrogram."""

    def __init__(self, d=192, heads=6):
        super().__init__()
        self.tts = CharTTS(d=d, heads=heads)
        self.spk_enc = SpeakerEncoder(d)
        self.spk_proj = nn.Linear(d, d)

    def forward(self, char_ids, refs):
        spk = self.spk_proj(self.spk_enc(refs))            # (B, d)
        # inject speaker into the frame queries (voice colours every output frame)
        B = char_ids.shape[0]
        mask = char_ids.eq(0)
        h = self.tts.emb(char_ids) + self.tts.cpos[None, :char_ids.shape[1]]
        mem = self.tts.encoder(h, src_key_padding_mask=mask)
        q = self.tts.frame_q[None].expand(B, -1, -1) + spk[:, None, :]
        dec = self.tts.decoder(q, mem, memory_key_padding_mask=mask)
        return self.tts.head(dec).transpose(1, 2)


def _refs(specs, w_target, vi, voices_words, rng, n_ref=4, device=DEV):
    """n_ref reference clips of voice vi (words != the query word, to force voice not word copy)."""
    pool = [w for w in voices_words if w != w_target and (w, vi) in specs]
    chosen = [pool[int(rng.integers(len(pool)))] for _ in range(n_ref)]
    return np.stack([specs[(w, vi)] for w in chosen])[:, None]   # (R,1,B,F)


def train(steps=6000, seed=0, device=DEV, batch=24, lr=3e-4, d=192):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    manifest, specs = load_bank()
    voices, words = manifest["voices"], manifest["words"]
    train_cells = [(wi, vi) for wi in range(len(words)) for vi in range(len(voices))
                   if (wi, vi) not in HOLDOUT_CELLS]
    model = MimicTTS(d=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        model.train()
        cells = [train_cells[int(rng.integers(len(train_cells)))] for _ in range(batch)]
        chars = torch.tensor([encode_chars(words[wi]) for wi, _ in cells], device=device)
        refs = torch.tensor(np.stack([_refs(specs, words[wi], vi, words, rng) for wi, vi in cells]),
                            dtype=torch.float32, device=device)   # (B,R,1,F,T)
        tgt = torch.stack([torch.tensor(specs[(words[wi], vi)], device=device) for wi, vi in cells])
        pred = model(chars, refs)
        loss = F.l1_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % max(1, steps // 8) == 0 or st == steps:
            print(f"  a5 {st}/{steps} L1 {loss.item():.4f}", flush=True)
    return model, manifest, specs


def _word_listener(specs, words, voices, device, steps=600, seed=0):
    """Frozen word classifier (verifies pronunciation survived voice transfer)."""
    X = torch.tensor(np.stack([specs[(w, vi)] for vi in range(len(voices)) for w in words]))[:, None].to(device)
    y = torch.tensor([words.index(w) for vi in range(len(voices)) for w in words], device=device)
    import torch.nn as nn
    net = nn.Sequential(nn.Conv2d(1, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                        nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                        nn.Conv2d(48, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)),
                        nn.Flatten(), nn.Linear(64 * 16, 128), nn.GELU(),
                        nn.Linear(128, len(words))).to(device)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i = torch.tensor(rng.integers(0, len(X), 48))
        loss = F.cross_entropy(net(X[i] + 0.01 * torch.randn_like(X[i])), y[i])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def _voice_centroids(model, specs, words, voices, device):
    """Target-voice embedding centroids from the model's own speaker encoder."""
    model.eval()
    cent = []
    with torch.no_grad():
        for vi in range(len(voices)):
            refs = torch.tensor(np.stack([specs[(w, vi)] for w in words[:16]]),
                                dtype=torch.float32, device=device)[None, :, None]
            cent.append(F.normalize(model.spk_proj(model.spk_enc(refs)), dim=-1)[0])
    return torch.stack(cent)


def evaluate(model, manifest, specs, device=DEV, seed=1):
    voices, words = manifest["voices"], manifest["words"]
    listener = _word_listener(specs, words, voices, device)
    centroids = _voice_centroids(model, specs, words, voices, device)
    rng = np.random.default_rng(seed)
    model.eval()
    word_ok = voice_ok = total = 0
    with torch.no_grad():
        for (wi, vi) in HOLDOUT_CELLS:
            refs = torch.tensor(_refs(specs, words[wi], vi, words, rng), dtype=torch.float32,
                                device=device)[None]
            chars = torch.tensor([encode_chars(words[wi])], device=device)
            spec = model(chars, refs)                       # (1, F, T)
            # word: re-analyze through Griffin-Lim audio then the frozen listener
            wav = griffin_lim(spec[0].cpu().numpy(), n_iter=40)
            relm = torch.tensor(stft_logmag(wav))[None, None].to(device)
            word_ok += int(listener(relm).argmax(-1).item() == wi)
            # voice: speaker embedding of the synthesized spec, nearest target voice
            emb = F.normalize(model.spk_proj(model.spk_enc(spec[None, None])), dim=-1)[0]
            voice_ok += int(int((centroids @ emb).argmax()) == vi)
            total += 1
    return {"holdout_cells": total, "word_acc": word_ok / total,
            "voice_match": voice_ok / total, "n_voices": len(voices)}


def run(steps=6000, seed=0, device=DEV):
    model, manifest, specs = train(steps=steps, seed=seed, device=device)
    ev = evaluate(model, manifest, specs, device=device)
    report = {"experiment": "a5_mimic", "steps": steps, **ev,
              "voice_chance": 1 / ev["n_voices"],
              "gate": ev["word_acc"] > 0.5 and ev["voice_match"] > 0.6}
    print(f"HELD-OUT (word x voice): word_acc {ev['word_acc']:.2f}  "
          f"voice_match {ev['voice_match']:.2f} (chance {1/ev['n_voices']:.2f})")
    print(f"gate: {report['gate']}", flush=True)
    return report, model, manifest, specs


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")):
        build_bank()
    manifest, specs = load_bank()
    voices, words = manifest["voices"], manifest["words"]
    assert (words[0], 0) in specs and specs[(words[0], 0)].shape == (N_BINS, N_FRAMES)
    assert len(HOLDOUT_CELLS) > 10
    m = MimicTTS(d=64, heads=4)
    rng = np.random.default_rng(0)
    refs = torch.tensor(_refs(specs, words[0], 0, words, rng), dtype=torch.float32)[None]
    out = m(torch.tensor([encode_chars(words[0])]), refs)
    assert out.shape == (1, N_BINS, N_FRAMES)
    model, mani, sp = train(steps=2, seed=0, device="cpu")
    ev = evaluate(model, mani, sp, device="cpu")
    assert "word_acc" in ev and "voice_match" in ev
    print("mimic selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--checkpoint", default="runs/a5_mimic.pt")
    ap.add_argument("--out", default="runs/a5_mimic.json")
    args = ap.parse_args(argv)
    if args.build:
        build_bank()
        return
    if args.selftest:
        selftest()
        return
    if args.train:
        report, model, _m, _s = run(steps=args.steps)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, args.checkpoint)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"saved -> {args.out}")
        return
    ap.error("choose --build / --selftest / --train")


if __name__ == "__main__":
    main()
