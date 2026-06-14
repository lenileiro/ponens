"""A-4 PRONOUNCE: character -> speech, generalizing to words NEVER HEARD.

A-3b memorized 12 whole-word spectrograms. "Understanding pronunciation" means a FACTORED,
compositional map from letters to sound -- so the model can say a word it was never trained on.
That is the FER question for speech generation: is letter->sound a reusable factor (UFR) or a
per-word lookup (FER)?

Model (tiny non-autoregressive TTS): char-sequence transformer encoder -> N_FRAMES learned frame
queries cross-attend over the character states (attention discovers the alignment, no explicit
durations) -> log-mel-ish spectrogram -> Griffin-Lim (vocoder.py, 25dB inversion). Trained L1 to
`say` oracle spectrograms on TRAIN words; tested on HELD-OUT words.

Verification (closed-form, no full ASR): synthesize a held-out word, and check its spectrogram's
nearest neighbour among ALL oracle word spectrograms is the right word -- "pronunciation
retrieval". If the model pronounces 'brick' (unseen) and that audio is closest to oracle 'brick',
it composed the pronunciation correctly.

  python -m thinking.pronounce --build
  python -m thinking.pronounce --selftest
  python -m thinking.pronounce --train --steps 6000 --out runs/a4_pronounce.json
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

DEV = get_device()
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                    "pronounce")
CHARS = "abcdefghijklmnopqrstuvwxyz"
C2I = {c: i + 1 for i, c in enumerate(CHARS)}              # 0 = pad
MAX_CHARS = 12

# ~180 common short English words (broad grapheme coverage); rendered by `say`.
WORDS = (
    "cat dog sun moon star tree leaf rock sand wind rain snow fire ice lake hill road gate "
    "bird fish frog bear wolf lion deer goat duck swan crow hawk mole seal crab moth wasp "
    "red blue green gold gray pink black white brown teal lime navy "
    "one two three four five six seven eight nine ten "
    "book pen desk lamp door wall roof floor stair chair table glass plate spoon fork bowl "
    "milk bread cake rice bean corn plum pear lime date fig kale "
    "run jump walk swim climb crawl dance sing read write draw paint build bake cook clean "
    "fast slow warm cold loud soft hard easy long wide deep tall thin "
    "north south east west left right near far high low "
    "king queen prince child friend baby uncle aunt niece "
    "brick steam cloud storm flame frost grass stone river ocean forest meadow valley "
    "happy angry brave quiet eager gentle clever honest "
    "apple lemon mango grape berry melon olive onion "
).split()
WORDS = tuple(dict.fromkeys(WORDS))                        # dedupe, keep order
HOLDOUT = tuple(WORDS[i] for i in range(3, len(WORDS), 7))  # ~1/7 held out for zero-shot test


def encode_chars(word):
    ids = [C2I[c] for c in word.lower() if c in C2I][:MAX_CHARS]
    return ids + [0] * (MAX_CHARS - len(ids))


def build_bank(root=ROOT, voice="Samantha"):
    os.makedirs(root, exist_ok=True)
    manifest = {"voice": voice, "words": list(WORDS)}
    for w in WORDS:
        p = os.path.join(root, f"{w}.wav")
        if not os.path.exists(p):
            subprocess.run(["say", "-v", voice, "-r", "180", "-o", p,
                            "--file-format=WAVE", f"--data-format=LEI16@{SR}", w],
                           check=True, timeout=60)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"pronounce bank: {len(WORDS)} words, voice {voice}, {len(HOLDOUT)} held out")
    return manifest


def load_bank(root=ROOT):
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    specs = {}
    for w in manifest["words"]:
        wv = wave.open(os.path.join(root, f"{w}.wav"))
        x = np.frombuffer(wv.readframes(wv.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        nz = np.where(np.abs(x) > 0.01)[0]
        if len(nz):
            x = x[max(0, nz[0] - 256):nz[-1] + 256]
        specs[w] = stft_logmag(x)                          # (N_BINS, N_FRAMES)
    return manifest, specs


class CharTTS(nn.Module):
    """char encoder -> learned frame queries cross-attend over char states -> spectrogram."""

    def __init__(self, n_frames=N_FRAMES, n_bins=N_BINS, d=192, heads=6, enc_layers=3,
                 dec_layers=3, n_speakers=0):
        super().__init__()
        self.emb = nn.Embedding(len(CHARS) + 1, d, padding_idx=0)
        self.cpos = nn.Parameter(torch.randn(MAX_CHARS, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers, enable_nested_tensor=False)
        self.frame_q = nn.Parameter(torch.randn(n_frames, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                         batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.spk = nn.Embedding(n_speakers, d) if n_speakers else None
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, n_bins))

    def forward(self, char_ids, speaker=None):
        B = char_ids.shape[0]
        mask = char_ids.eq(0)
        h = self.emb(char_ids) + self.cpos[None, :char_ids.shape[1]]
        mem = self.encoder(h, src_key_padding_mask=mask)
        q = self.frame_q[None].expand(B, -1, -1)
        if self.spk is not None and speaker is not None:
            q = q + self.spk(speaker)[:, None, :]
        dec = self.decoder(q, mem, memory_key_padding_mask=mask)
        return self.head(dec).transpose(1, 2)              # (B, n_bins, n_frames)


def _tensor(words, device):
    return torch.tensor([encode_chars(w) for w in words], dtype=torch.long, device=device)


def train(steps=6000, seed=0, device=DEV, batch=32, lr=3e-4, d=192):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _manifest, specs = load_bank()
    train_words = [w for w in WORDS if w not in HOLDOUT]
    targets = {w: torch.tensor(specs[w], device=device) for w in WORDS}
    model = CharTTS(d=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        model.train()
        bw = [train_words[int(rng.integers(len(train_words)))] for _ in range(batch)]
        pred = model(_tensor(bw, device))
        tgt = torch.stack([targets[w] for w in bw])
        loss = F.l1_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % max(1, steps // 8) == 0 or st == steps:
            print(f"  a4 {st}/{steps} L1 {loss.item():.4f}", flush=True)
    return model, specs


def pronunciation_retrieval(model, specs, words, device, topk=(1, 3)):
    """Synthesize each word; nearest oracle-spectrogram neighbour among ALL words. Held-out words
    test zero-shot pronunciation: is the model's audio closest to the RIGHT oracle word?"""
    model.eval()
    bank_words = list(WORDS)
    bank = torch.stack([torch.tensor(specs[w], device=device) for w in bank_words]).flatten(1)
    bank = F.normalize(bank, dim=-1)
    hits = {k: 0 for k in topk}
    with torch.no_grad():
        pred = model(_tensor(words, device)).flatten(1)
        pred = F.normalize(pred, dim=-1)
        sims = pred @ bank.t()                              # (len(words), n_bank)
        order = sims.argsort(-1, descending=True)
        for i, w in enumerate(words):
            gold = bank_words.index(w)
            ranked = order[i].tolist()
            for k in topk:
                hits[k] += int(gold in ranked[:k])
    return {f"top{k}": hits[k] / len(words) for k in topk}


def synth(model, words, out_dir, device=DEV):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for w in words:
            lm = model(_tensor([w], device))[0].cpu().numpy()
            write_wav(os.path.join(out_dir, f"{w}_pron.wav"), griffin_lim(lm))
            print(f"  {w} -> {out_dir}/{w}_pron.wav")


def run(steps=6000, seed=0, device=DEV):
    model, specs = train(steps=steps, seed=seed, device=device)
    train_words = [w for w in WORDS if w not in HOLDOUT]
    seen = pronunciation_retrieval(model, specs, train_words, device)
    held = pronunciation_retrieval(model, specs, list(HOLDOUT), device)
    report = {"experiment": "a4_pronounce", "steps": steps, "n_words": len(WORDS),
              "n_holdout": len(HOLDOUT), "chance_top1": 1 / len(WORDS),
              "seen_retrieval": seen, "holdout_retrieval": held,
              "gate": held["top3"] > 0.5}
    print(f"SEEN words   : {seen}")
    print(f"HELD-OUT words (never heard): {held}   chance top1 {1/len(WORDS):.3f}")
    print(f"gate (held-out top3 > 0.5): {report['gate']}", flush=True)
    return report, model


def selftest():
    if not os.path.exists(os.path.join(ROOT, "manifest.json")):
        build_bank()
    _manifest, specs = load_bank()
    assert len(specs) == len(WORDS) and specs[WORDS[0]].shape == (N_BINS, N_FRAMES)
    assert encode_chars("cat")[:3] == [C2I["c"], C2I["a"], C2I["t"]]
    assert set(HOLDOUT).issubset(set(WORDS)) and 5 < len(HOLDOUT) < len(WORDS)
    m = CharTTS(d=64, heads=4, enc_layers=1, dec_layers=1)
    out = m(_tensor(["cat", "dog"], "cpu"))
    assert out.shape == (2, N_BINS, N_FRAMES)
    model, _ = train(steps=3, seed=0, device="cpu")
    r = pronunciation_retrieval(model, specs, list(HOLDOUT)[:4], "cpu")
    assert "top1" in r and "top3" in r
    print("pronounce selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--synth", default="")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--checkpoint", default="runs/a4_pronounce.pt")
    ap.add_argument("--out", default="runs/a4_pronounce.json")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    args = ap.parse_args(argv)
    if args.build:
        build_bank()
        return
    if args.selftest:
        selftest()
        return
    if args.train:
        report, model = run(steps=args.steps)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, args.checkpoint)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        # synth a few held-out words so they can be played
        synth(model, list(HOLDOUT)[:6], args.synth_out)
        print(f"saved -> {args.out}, {args.checkpoint}")
        return
    if args.synth:
        model = CharTTS().to(DEV)
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEV)["state_dict"])
        synth(model, [w.strip() for w in args.synth.split(",")], args.synth_out)
        return
    ap.error("choose --build / --selftest / --train / --synth")


if __name__ == "__main__":
    main()
