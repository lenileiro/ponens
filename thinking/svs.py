"""Singing-Voice Synthesis: the model LEARNS to sing from real singing, conditioned on MELODY.

The earlier `prosody.sing` was a DSP trick (re-pitch spoken audio onto a melody I typed in) -- the
model had no concept of singing. This is the real thing: a melody-conditioned acoustic model trained
on the Children's Song Dataset (CSD, 50 English songs, one singer, MIDI + phoneme-aligned lyrics).

Inputs per note (from the score): the syllable's PHONEMES + the note's PITCH (MIDI) + its DURATION.
The model is the same autoregressive Transformer that made speech intelligible (tts_ar), with one
addition: a PITCH EMBEDDING added to the phoneme encoder states. So pitch is a learned conditioning
the model realizes as an F0 contour (with vibrato/sustain) -- it learns *how to sing* a note from
real sung audio, instead of having speech mechanically re-pitched. Same 513-bin log-mag target +
realvoice vocoder.

Objective check (no human ear): synthesize a HELD-OUT phrase, estimate the dominant F0 per note from
the audio, and compare to the score's MIDI note -> pitch accuracy (in tune?).

  python -m thinking.svs --selftest
  python -m thinking.svs --fetch
  python -m thinking.svs --train --steps 30000 --out runs/svs.json   (GPU)
"""
import argparse
import json
import os
import zipfile
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import get_device
from .realvoice import SR, N_BINS
from .tts import MAX_T, spec_of, guided_attention_loss
from .tts_ar import DecoderLayer, sinusoidal, R

DEV = get_device()
CSD_URL = "https://zenodo.org/api/records/4785016/files/CSD.zip/content"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "csd")
MAX_NOTES = 40                          # notes per training phrase (segment songs into phrases)
PAD_PITCH = 0                           # 0 = pad/rest; real MIDI pitches are 1..127


# ---- data: download CSD, parse MIDI + phoneme lyrics + audio -> phrase items --------------------
def _parse_midi(path):
    """Monophonic MIDI -> [(pitch, start_sec, end_sec)] in order."""
    import mido
    mid = mido.MidiFile(path)
    tempo = 500000
    notes, t, on = [], 0.0, {}
    for msg in mido.merge_tracks(mid.tracks):
        t += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            on[msg.note] = t
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in on:
                notes.append((msg.note, on.pop(msg.note), t))
    return sorted(notes, key=lambda n: n[1])


def fetch(root=ROOT):
    import soundfile as sf
    os.makedirs(root, exist_ok=True)
    zp = os.path.join(root, "CSD.zip")
    if not os.path.exists(zp):
        print(f"downloading CSD (1.85GB) -> {zp} ...", flush=True)
        urllib.request.urlretrieve(CSD_URL, zp)
    z = zipfile.ZipFile(zp)
    # locate english song stems present in all of mid/, txt/, wav/
    members = z.namelist()
    def pick(folder, ext):
        return {os.path.splitext(os.path.basename(m))[0]: m for m in members
                if f"/english/{folder}/" in m.lower() and m.lower().endswith(ext)}
    mids, txts, wavs = pick("mid", ".mid"), pick("txt", ".txt"), pick("wav", ".wav")
    stems = sorted(set(mids) & set(txts) & set(wavs))
    print(f"CSD english songs: {len(stems)}", flush=True)
    os.makedirs(os.path.join(root, "spec"), exist_ok=True)
    import tempfile, librosa
    items, phon_set = [], set()
    for stem in stems:
        with tempfile.TemporaryDirectory() as td:
            mp = z.extract(mids[stem], td); tp = z.extract(txts[stem], td); wp = z.extract(wavs[stem], td)
            notes = _parse_midi(mp)
            sylls = open(tp).read().split()                       # phoneme syllables (one per note)
            n = min(len(notes), len(sylls))
            notes, sylls = notes[:n], sylls[:n]
            wav, sr = sf.read(wp)
            if wav.ndim > 1:
                wav = wav.mean(1)
            if sr != SR:
                wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)
            # segment into phrases of <= MAX_NOTES notes that fit in MAX_T frames (~4s)
            for s in range(0, n, MAX_NOTES):
                seg = list(zip(notes[s:s + MAX_NOTES], sylls[s:s + MAX_NOTES]))
                if not seg:
                    continue
                t0, t1 = seg[0][0][1], seg[-1][0][2]
                if t1 - t0 < 0.4 or (t1 - t0) > MAX_T * 256 / SR:
                    if (t1 - t0) > MAX_T * 256 / SR:
                        seg = seg[: max(1, int(len(seg) * (MAX_T * 256 / SR) / (t1 - t0)))]
                        t1 = seg[-1][0][2]
                    if t1 - t0 < 0.4:
                        continue
                toks, pits = [], []
                for (pitch, ns, ne), syl in seg:
                    ph = syl.split("_")
                    phon_set.update(ph)
                    toks += ph; pits += [pitch] * len(ph)
                a0, a1 = int(t0 * SR), int(t1 * SR)
                items.append({"stem": stem, "toks": toks, "pitches": pits,
                              "wav": wav[a0:a1].astype(np.float32)})
    # phoneme vocab + persist specs
    vocab = {p: i + 1 for i, p in enumerate(sorted(phon_set))}     # 0 = pad
    clips = []
    for k, it in enumerate(items):
        cid = f"c{k:05d}"
        np.save(os.path.join(root, "spec", cid + ".npy"), it["wav"])
        clips.append({"id": cid, "toks": [vocab[t] for t in it["toks"]], "pitches": it["pitches"]})
    json.dump({"vocab": vocab, "clips": clips}, open(os.path.join(root, "manifest.json"), "w"))
    print(f"parsed {len(clips)} phrases, {len(vocab)} phonemes -> {root}/manifest.json", flush=True)


def load(root=ROOT):
    man = json.load(open(os.path.join(root, "manifest.json")))
    out = []
    for c in man["clips"]:
        p = os.path.join(root, "spec", c["id"] + ".npy")
        if os.path.exists(p):
            out.append((c["toks"], c["pitches"], np.load(p)))
    return out, man["vocab"]


# ---- model: phoneme + pitch -> sung spectrogram (AR Transformer, melody-conditioned) ------------
class SingTTS(nn.Module):
    def __init__(self, n_phon, d=384, n_bins=N_BINS, layers=6, heads=6, r=R):
        super().__init__()
        self.d = d; self.r = r; self.n_bins = n_bins
        self.cfg = {"n_phon": n_phon, "d": d, "layers": layers, "heads": heads, "r": r}
        self.phon = nn.Embedding(n_phon + 1, d, padding_idx=0)
        self.pitch = nn.Embedding(128, d)                          # MIDI pitch conditioning
        self.enc_conv = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(d, d, 5, padding=2), nn.GELU())
        self.prenet = nn.Sequential(nn.Linear(n_bins, 256), nn.ReLU(), nn.Linear(256, d), nn.ReLU())
        self.layers = nn.ModuleList([DecoderLayer(d, heads) for _ in range(layers)])
        self.out = nn.Linear(d, r * n_bins)
        self.stop = nn.Linear(d, 1)
        pn, ch = [], 256
        for i in range(5):
            ci, co = (n_bins if i == 0 else ch), (n_bins if i == 4 else ch)
            pn += [nn.Conv1d(ci, co, 5, padding=2)]
            if i < 4:
                pn += [nn.BatchNorm1d(co), nn.Tanh(), nn.Dropout(0.1)]
        self.postnet = nn.Sequential(*pn)

    def memory(self, toks, pitches):
        h = (self.phon(toks) + self.pitch(pitches.clamp(0, 127))).transpose(1, 2)   # (B,L,d)->(B,d,L)
        return self.enc_conv(h).transpose(1, 2)                    # (B,L,d) pitch-aware phoneme states

    def prenet_do(self, x):
        x = F.dropout(F.relu(self.prenet[0](x)), 0.5, training=True)
        return F.dropout(F.relu(self.prenet[2](x)), 0.5, training=True)

    def _run(self, group_in, mem, mem_pad):
        Tg = group_in.shape[1]
        x = self.prenet_do(group_in) + sinusoidal(Tg, self.d, group_in.device)[None]
        causal = torch.triu(torch.full((Tg, Tg), float("-inf"), device=group_in.device), 1)
        ws = []
        for layer in self.layers:
            x, w = layer(x, mem, causal, mem_pad); ws.append(w)
        return x, ws

    def decode(self, toks, pitches, mel):
        B, nb, T = mel.shape; Tg = T // self.r
        mem = self.memory(toks, pitches); mem_pad = toks.eq(0)
        last = mel[:, :, self.r - 1::self.r]
        start = torch.zeros(B, nb, 1, device=toks.device)
        group_in = torch.cat([start, last[:, :, :-1]], 2).transpose(1, 2)
        if self.training:
            group_in = group_in + 0.25 * torch.randn_like(group_in)
        x, ws = self._run(group_in, mem, mem_pad)
        coarse = self.out(x).reshape(B, Tg, self.r, nb).permute(0, 3, 1, 2).reshape(B, nb, Tg * self.r)
        return coarse, coarse + self.postnet(coarse), self.stop(x).squeeze(-1), ws

    @torch.no_grad()
    def infer(self, toks, pitches, max_T=MAX_T, stop_thresh=0.5):
        self.eval()
        mem = self.memory(toks, pitches); mem_pad = toks.eq(0); B = toks.shape[0]
        group_in = torch.zeros(B, 1, self.n_bins, device=toks.device); outs = []
        for k in range(max_T // self.r):
            x, _ = self._run(group_in, mem, mem_pad)
            nxt = self.out(x[:, -1]).reshape(B, self.r, self.n_bins); outs.append(nxt)
            group_in = torch.cat([group_in, nxt[:, -1:, :]], 1)
            if k > 4 and torch.sigmoid(self.stop(x[:, -1])).max().item() > stop_thresh:
                break
        coarse = torch.cat(outs, 1).transpose(1, 2)
        return coarse + self.postnet(coarse)


def _batch(cache, rng, batch, device):
    specs, toks_all, pit_all = cache
    idx = [int(rng.integers(len(specs))) for _ in range(batch)]
    toks = [toks_all[j] for j in idx]; pits = [pit_all[j] for j in idx]; sp = [specs[j] for j in idx]
    Lc = max(len(t) for t in toks); Tt = max(s.shape[1] for s in sp)
    Tt = ((Tt + R - 1) // R) * R; Tg = Tt // R
    tk = torch.zeros(batch, Lc, dtype=torch.long, device=device)
    pt = torch.zeros(batch, Lc, dtype=torch.long, device=device)
    mel = torch.zeros(batch, N_BINS, Tt, device=device)
    stop = torch.zeros(batch, Tg, device=device)
    tl = torch.zeros(batch, dtype=torch.long, device=device)
    gl = torch.zeros(batch, dtype=torch.long, device=device)
    for b, (t, p, s) in enumerate(zip(toks, pits, sp)):
        tk[b, :len(t)] = torch.tensor(t, device=device); tl[b] = len(t)
        pt[b, :len(p)] = torch.tensor(p, device=device)
        mel[b, :, :s.shape[1]] = s.to(device)
        g = (s.shape[1] + R - 1) // R; gl[b] = g; stop[b, g - 1:] = 1.0
    return tk, pt, mel, stop, tl, gl


def build_cache(data):
    specs = [spec_of(w, "cpu")[:, :MAX_T].contiguous() for _, _, w in data]   # spec_of returns a tensor
    toks = [t for t, _, _ in data]; pits = [p for _, p, _ in data]
    return specs, toks, pits


def train(steps=30000, seed=0, device=DEV, batch=64, lr=4e-4, ckpt_path=None, d=384, layers=6, heads=6):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data, vocab = load(); split = int(len(data) * 0.95); td = data[:split]
    print(f"caching {len(td)} singing phrases ...", flush=True)
    cache = build_cache(td)
    model = SingTTS(len(vocab), d=d, layers=layers, heads=heads).to(device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        tk, pt, mel, stop, tl, gl = _batch(cache, rng, batch, device)
        coarse, refined, stop_logit, aligns = model.decode(tk, pt, mel)
        mmask = (torch.arange(mel.shape[2], device=device)[None] < (gl * R)[:, None]).float()
        l_mel = sum((F.l1_loss(p, mel, reduction="none").mean(1) * mmask).sum() / mmask.sum()
                    for p in (coarse, refined))
        l_stop = F.binary_cross_entropy_with_logits(stop_logit, stop)
        l_ga = sum(guided_attention_loss(a[b:b + 1], int(tl[b]), int(gl[b]))
                   for a in aligns for b in range(batch)) / (batch * len(aligns))
        (l_mel + l_stop + 5.0 * l_ga).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        if st % max(1, steps // 20) == 0 or st == steps:
            print(f"  svs {st}/{steps} mel {l_mel.item():.3f} ga {l_ga.item():.4f} "
                  f"focus {float(aligns[-1].max(-1).values.mean().detach()):.3f}", flush=True)
        if st % max(1, steps // 5) == 0 and ckpt_path:
            torch.save({"state_dict": model.state_dict(), "config": model.cfg, "vocab": vocab}, ckpt_path)
    return model, data, vocab


# ---- objective pitch-accuracy eval (is it in tune?) ---------------------------------------------
def _dom_f0(seg, sr=SR):
    if len(seg) < 256:
        return 0.0
    w = seg * np.hanning(len(seg))
    mag = np.abs(np.fft.rfft(w)); freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    band = (freqs > 80) & (freqs < 1000)
    return float(freqs[band][np.argmax(mag[band])]) if band.any() else 0.0


def _midi_of_hz(hz):
    return 69 + 12 * np.log2(hz / 440.0) if hz > 0 else 0


def evaluate(model, data, vocab, device=DEV, n=40, seed=1):
    from .realvoice import Vocos
    voc = Vocos(d=512).to(device)
    if os.path.exists("runs/realvoice.pt"):
        voc.load_state_dict(torch.load("runs/realvoice.pt", map_location=device)["state_dict"]); voc.eval()
    rng = np.random.default_rng(seed); ev = data[int(len(data) * 0.95):]
    model.eval(); hits = tot = 0
    with torch.no_grad():
        for _ in range(min(n, len(ev))):
            toks, pits, _ = ev[int(rng.integers(len(ev)))]
            tk = torch.tensor([toks], device=device); pt = torch.tensor([pits], device=device)
            spec = model.infer(tk, pt)
            wav = voc(spec)[0].cpu().numpy()
            # split audio evenly across notes, check dominant F0 maps near the target pitch
            uniq = []
            for p in pits:
                if not uniq or uniq[-1][0] != p:
                    uniq.append([p, 1])
                else:
                    uniq[-1][1] += 1
            segn = max(1, len(wav) // max(1, len(uniq)))
            for i, (p, _) in enumerate(uniq):
                seg = wav[i * segn:(i + 1) * segn]
                m = _midi_of_hz(_dom_f0(seg))
                if m and abs(((m - p + 6) % 12) - 6) <= 1.5:        # within ~1 semitone (octave-folded)
                    hits += 1
                tot += 1
    return {"pitch_acc": hits / max(1, tot), "n_notes": tot}


def run(steps=30000, seed=0, device=DEV, ckpt_path="runs/svs.pt", batch=64, d=384, layers=6, heads=6):
    model, data, vocab = train(steps=steps, seed=seed, device=device, ckpt_path=ckpt_path,
                               batch=batch, d=d, layers=layers, heads=heads)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg, "vocab": vocab}, ckpt_path)
    ev = evaluate(model, data, vocab, device=device)
    report = {"experiment": "svs_csd_english", "sr": SR, "steps": steps, **model.cfg, **ev}
    print(f"\nSINGING pitch accuracy {ev['pitch_acc']:.3f} over {ev['n_notes']} notes", flush=True)
    return report, model


def selftest():
    torch.manual_seed(0)
    m = SingTTS(n_phon=30, d=64, layers=2, heads=4)
    toks = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
    pits = torch.tensor([[60, 60, 62, 64, 0], [67, 67, 69, 0, 0]])
    mel = torch.randn(2, N_BINS, 24)
    coarse, refined, stoplg, aligns = m.decode(toks, pits, mel)
    assert coarse.shape == (2, N_BINS, 24) and aligns[-1].shape[0] == 2
    out = m.infer(toks[:1], pits[:1], max_T=12); assert out.shape[1] == N_BINS
    have = os.path.exists(os.path.join(ROOT, "manifest.json"))
    print(f"svs selftest OK (data present: {have})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--out", default="runs/svs.json")
    ap.add_argument("--checkpoint", default="runs/svs.pt")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest(); return
    if args.fetch:
        fetch(); return
    if args.train:
        report, _ = run(steps=args.steps, batch=args.batch, d=args.dim, layers=args.layers,
                        heads=args.heads, ckpt_path=args.checkpoint)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"saved -> {args.out}")
        return
    ap.error("choose --selftest / --fetch / --train")


if __name__ == "__main__":
    main()
