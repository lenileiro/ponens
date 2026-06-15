"""Text-to-speech: text -> spectrogram -> waveform, reusing the trained realvoice vocoder.

The pieces are all here now: a natural neural vocoder (realvoice, LJSpeech 22kHz log-mag -> wav)
and the char->acoustic modeling from pronounce. TTS adds the hard middle: aligning a character
sequence to acoustic frames. The robust single-speaker recipe is a Tacotron-style AUTOREGRESSIVE
decoder with LOCATION-SENSITIVE attention, trained with a GUIDED-ATTENTION loss (a diagonal prior
that forces monotonic text->audio alignment -- the thing that makes small-scale Tacotron actually
align). Acoustic model predicts the 513-bin log-mag spectrogram realvoice's vocoder consumes, so
synthesis is: text -> spectrogram (this model) -> realvoice Vocos -> natural waveform.

Trained on LJSpeech (the single-speaker TTS standard; transcripts from metadata.csv).

  python -m thinking.tts --fetch
  python -m thinking.tts --selftest
  python -m thinking.tts --train --steps 60000 --out runs/tts.json   (GPU)
"""
import argparse
import json
import os
import tarfile
import urllib.request
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .realvoice import SR, N_FFT, HOP, N_BINS, logmag_feat

DEV = get_device()
URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ljtts")
CHARS = " abcdefghijklmnopqrstuvwxyz',.?!-"
C2I = {c: i + 1 for i, c in enumerate(CHARS)}              # 0 = pad
MAX_T = 360                                                # spectrogram frames cap (~4s)
MAX_CH = 180


def encode_text(t):
    t = t.lower()
    return [C2I[c] for c in t if c in C2I][:MAX_CH]


def fetch(n_clips=4000, root=ROOT, byte_cap_gb=0.9):
    """Stream LJSpeech; parse metadata.csv (transcripts) + extract matching wavs as (text, wav)."""
    import soundfile as sf
    os.makedirs(root, exist_ok=True)
    seen = [0]; cap = int(byte_cap_gb * 1e9)

    class _C:
        def __init__(s, f): s.f = f
        def read(s, n):
            b = s.f.read(n); seen[0] += len(b); return b

    print(f"streaming {URL} (cap {byte_cap_gb}GB, {n_clips} clips)...")
    req = urllib.request.Request(URL, headers={"User-Agent": "ponens/1.0"})
    texts = {}
    clips = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        tar = tarfile.open(fileobj=_C(resp), mode="r|bz2")
        for m in tar:
            if seen[0] > cap or len(clips) >= n_clips:
                break
            if m.name.endswith("metadata.csv"):
                for line in tar.extractfile(m).read().decode("utf8").splitlines():
                    parts = line.split("|")
                    if len(parts) >= 3:
                        texts[parts[0]] = parts[2]          # normalized transcript
            elif m.name.endswith(".wav"):
                fid = os.path.basename(m.name)[:-4]
                if fid not in texts:
                    continue
                x, sr = sf.read(io.BytesIO(tar.extractfile(m).read()))
                if sr != SR or len(x) < SR // 2 or len(x) > SR * 9:
                    continue
                np.save(os.path.join(root, fid + ".npy"), x.astype(np.float32))
                clips.append({"id": fid, "text": texts[fid]})
    json.dump({"clips": clips}, open(os.path.join(root, "manifest.json"), "w"))
    print(f"fetched {len(clips)} (text,wav) pairs, {seen[0]/1e9:.2f}GB")
    return len(clips)


def load(root=ROOT):
    man = json.load(open(os.path.join(root, "manifest.json")))
    out = []
    for c in man["clips"]:
        p = os.path.join(root, c["id"] + ".npy")
        if os.path.exists(p):
            out.append((c["text"], np.load(p)))
    return out


def spec_of(wav, device):
    return logmag_feat(torch.tensor(wav[None], dtype=torch.float32, device=device))[0]   # (513, T)


class CharEncoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.emb = nn.Embedding(len(CHARS) + 1, d, padding_idx=0)
        self.conv = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(d, d, 5, padding=2), nn.GELU())
        self.lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)

    def forward(self, ids):
        h = self.emb(ids).transpose(1, 2)
        h = self.conv(h).transpose(1, 2)
        return self.lstm(h)[0]                              # (B, L, d)


class Tacotron(nn.Module):
    """AR spectrogram decoder with location-sensitive attention over character encodings."""

    def __init__(self, d=256, n_bins=N_BINS):
        super().__init__()
        self.enc = CharEncoder(d)
        self.prenet = nn.Sequential(nn.Linear(n_bins, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU())
        self.attn_rnn = nn.LSTMCell(128 + d, d)
        self.attn_w = nn.Linear(d, d, bias=False)          # query
        self.attn_v = nn.Linear(d, d, bias=False)          # keys
        self.attn_loc = nn.Conv1d(1, d, 31, padding=15)    # location features (prev attention)
        self.attn_score = nn.Linear(d, 1, bias=False)
        self.dec_rnn = nn.LSTMCell(d + d, d)
        self.out = nn.Linear(d, n_bins)
        self.stop = nn.Linear(d, 1)
        self.d = d

    def forward(self, ids, mel_in):
        """Teacher-forced: ids (B, L), mel_in (B, n_bins, T) shifted -> predict (B, n_bins, T)."""
        B, _, T = mel_in.shape
        keys = self.enc(ids)                               # (B, L, d)
        kproj = self.attn_v(keys)
        mask = ids.eq(0)
        dev = ids.device
        h_a = torch.zeros(B, self.d, device=dev); c_a = torch.zeros(B, self.d, device=dev)
        h_d = torch.zeros(B, self.d, device=dev); c_d = torch.zeros(B, self.d, device=dev)
        attn_w = torch.zeros(B, keys.shape[1], device=dev)
        ctx = torch.zeros(B, self.d, device=dev)
        outs, stops, aligns = [], [], []
        for t in range(T):
            pre = self.prenet(mel_in[:, :, t])
            h_a, c_a = self.attn_rnn(torch.cat([pre, ctx], -1), (h_a, c_a))
            loc = self.attn_loc(attn_w[:, None]).transpose(1, 2)   # (B, L, d)
            score = self.attn_score(torch.tanh(self.attn_w(h_a)[:, None] + kproj + loc)).squeeze(-1)
            score = score.masked_fill(mask, -1e9)
            attn_w = F.softmax(score, -1)
            ctx = (attn_w[:, :, None] * keys).sum(1)
            h_d, c_d = self.dec_rnn(torch.cat([h_a, ctx], -1), (h_d, c_d))
            outs.append(self.out(h_d)); stops.append(self.stop(h_d)); aligns.append(attn_w)
        return (torch.stack(outs, -1), torch.stack(stops, -1).squeeze(1),
                torch.stack(aligns, 1))                     # mel, stop_logits, aligns (B,T,L)

    @torch.no_grad()
    def infer(self, ids, max_T=MAX_T):
        B = ids.shape[0]
        keys = self.enc(ids); kproj = self.attn_v(keys); mask = ids.eq(0)
        dev = ids.device
        h_a = torch.zeros(B, self.d, device=dev); c_a = torch.zeros(B, self.d, device=dev)
        h_d = torch.zeros(B, self.d, device=dev); c_d = torch.zeros(B, self.d, device=dev)
        attn_w = torch.zeros(B, keys.shape[1], device=dev); ctx = torch.zeros(B, self.d, device=dev)
        frame = torch.zeros(B, N_BINS, device=dev)
        outs = []
        for t in range(max_T):
            pre = self.prenet(frame)
            h_a, c_a = self.attn_rnn(torch.cat([pre, ctx], -1), (h_a, c_a))
            loc = self.attn_loc(attn_w[:, None]).transpose(1, 2)
            score = self.attn_score(torch.tanh(self.attn_w(h_a)[:, None] + kproj + loc)).squeeze(-1)
            attn_w = F.softmax(score.masked_fill(mask, -1e9), -1)
            ctx = (attn_w[:, :, None] * keys).sum(1)
            h_d, c_d = self.dec_rnn(torch.cat([h_a, ctx], -1), (h_d, c_d))
            frame = self.out(h_d); outs.append(frame)
            if torch.sigmoid(self.stop(h_d)).item() > 0.5 and t > 4:
                break
        return torch.stack(outs, -1)                        # (B, n_bins, T)


def guided_attention_loss(aligns, ids_len, mel_len):
    """Penalize off-diagonal attention -> forces monotonic alignment (small-scale Tacotron fix)."""
    B, T, L = aligns.shape
    t = torch.arange(T, device=aligns.device)[None, :, None] / max(1, mel_len)
    l = torch.arange(L, device=aligns.device)[None, None, :] / max(1, ids_len)
    W = 1 - torch.exp(-((t - l) ** 2) / (2 * 0.2 ** 2))    # diagonal prior
    return (aligns * W).mean()


def _batch(data, rng, batch, device):
    items = [data[int(rng.integers(len(data)))] for _ in range(batch)]
    specs = [spec_of(w, device)[:, :MAX_T] for _, w in items]
    ids = [encode_text(t) for t, _ in items]
    Lc = max(len(i) for i in ids); Tt = max(s.shape[1] for s in specs)
    idt = torch.zeros(batch, Lc, dtype=torch.long, device=device)
    mel = torch.zeros(batch, N_BINS, Tt, device=device)
    stop = torch.zeros(batch, Tt, device=device)
    for b, (i, s) in enumerate(zip(ids, specs)):
        idt[b, :len(i)] = torch.tensor(i, device=device)
        mel[b, :, :s.shape[1]] = s
        stop[b, s.shape[1] - 1:] = 1.0                     # stop at end of real frames
    return idt, mel, stop


def train(steps=60000, seed=0, device=DEV, batch=16, lr=3e-4):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = load()
    split = int(len(data) * 0.95); train_data = data[:split]
    model = Tacotron().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        idt, mel, stop = _batch(train_data, rng, batch, device)
        mel_in = F.pad(mel, (1, 0))[:, :, :-1]             # shift for teacher forcing
        pred, stop_logit, aligns = model(idt, mel_in)
        l_mel = F.l1_loss(pred, mel)
        l_stop = F.binary_cross_entropy_with_logits(stop_logit, stop)
        l_ga = guided_attention_loss(aligns, idt.shape[1], mel.shape[2])
        (l_mel + l_stop + 2.0 * l_ga).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if st % max(1, steps // 12) == 0 or st == steps:
            print(f"  tts {st}/{steps} mel {l_mel.item():.3f} stop {l_stop.item():.3f} "
                  f"ga {l_ga.item():.4f}", flush=True)
    return model, data


def evaluate(model, data, device=DEV, n=200, seed=1):
    """Held-out teacher-forced spectral L1 + attention diagonality (did it align?)."""
    rng = np.random.default_rng(seed)
    eval_data = data[int(len(data) * 0.95):]
    model.eval()
    errs, diag = [], []
    with torch.no_grad():
        for _ in range(min(n, len(eval_data) * 2)):
            idt, mel, _ = _batch(eval_data, rng, 1, device)
            mel_in = F.pad(mel, (1, 0))[:, :, :-1]
            pred, _, aligns = model(idt, mel_in)
            errs.append(float(F.l1_loss(pred, mel)))
            # diagonality: fraction of attention mass on the argmax monotonic path proxy
            diag.append(float(aligns.max(-1).values.mean()))
    return {"heldout_spec_l1": float(np.mean(errs)), "attention_focus": float(np.mean(diag))}


def synth(model, texts, out_dir, device=DEV):
    """text -> spectrogram -> realvoice vocoder -> wav."""
    from .realvoice import Vocos, griffin_lim, write_wav
    os.makedirs(out_dir, exist_ok=True)
    voc = None
    if os.path.exists("runs/realvoice.pt"):
        voc = Vocos(d=512).to(device)
        voc.load_state_dict(torch.load("runs/realvoice.pt", map_location=device)["state_dict"])
        voc.eval()
    model.eval()
    for i, t in enumerate(texts):
        ids = torch.tensor([encode_text(t)], device=device)
        spec = model.infer(ids)[0].cpu().numpy()
        if voc is not None:
            with torch.no_grad():
                wav = voc(torch.tensor(spec[None], device=device))[0].cpu().numpy()
        else:
            wav = griffin_lim(spec)
        write_wav(os.path.join(out_dir, f"tts{i}.wav"), wav)
        print(f"  tts{i}: \"{t[:50]}\" -> {out_dir}/tts{i}.wav ({spec.shape[1]} frames)")


def run(steps=60000, seed=0, device=DEV, save_dir="data/synth"):
    model, data = train(steps=steps, seed=seed, device=device)
    ev = evaluate(model, data, device=device)
    report = {"experiment": "tts_tacotron_ljspeech", "sr": SR, "steps": steps, **ev,
              "aligned": ev["attention_focus"] > 0.4}
    print(f"\nheld-out spectral L1 {ev['heldout_spec_l1']:.3f}  attention_focus {ev['attention_focus']:.3f}")
    synth(model, ["the quick brown fox jumps over the lazy dog.",
                  "hello, this is a test of the speech system.",
                  "she sells sea shells by the sea shore."], save_dir, device=device)
    return report, model


def selftest():
    m = Tacotron()
    a, b = encode_text("hello world"), encode_text("a test.")
    L = max(len(a), len(b))
    ids = torch.tensor([a + [0] * (L - len(a)), b + [0] * (L - len(b))])
    mel = torch.randn(2, N_BINS, 20)
    pred, stop, aligns = m(ids, mel)
    assert pred.shape == (2, N_BINS, 20) and aligns.shape == (2, 20, ids.shape[1]), (pred.shape, aligns.shape)
    out = m.infer(ids[:1], max_T=15)
    assert out.shape[0] == 1 and out.shape[1] == N_BINS
    ga = guided_attention_loss(aligns, ids.shape[1], 20)
    assert torch.isfinite(ga)
    if os.path.exists(os.path.join(ROOT, "manifest.json")):
        data = load(); assert data and isinstance(data[0][0], str)
        m2, d = train(steps=2, seed=0, device="cpu", batch=2)
        ev = evaluate(m2, d, device="cpu", n=3)
        assert "heldout_spec_l1" in ev
    print("tts selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--n-clips", type=int, default=4000, dest="n_clips")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/tts.json")
    ap.add_argument("--checkpoint", default="runs/tts.pt")
    ap.add_argument("--synth-out", default="data/synth", dest="synth_out")
    ap.add_argument("--say", default="")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch(n_clips=args.n_clips); return
    if args.selftest:
        selftest(); return
    if args.train:
        report, model = run(steps=args.steps, seed=args.seed, save_dir=args.synth_out)
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, args.checkpoint)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"saved -> {args.out}")
        return
    if args.say:
        model = Tacotron().to(DEV)
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEV)["state_dict"])
        synth(model, [args.say], args.synth_out)
        return
    ap.error("choose --fetch / --selftest / --train / --say")


if __name__ == "__main__":
    main()
